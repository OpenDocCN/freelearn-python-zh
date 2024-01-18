# 规划我们应用程序的扩展

这个应用程序真的很受欢迎！经过一些初步测试和定位，数据录入人员现在已经使用您的新表单几个星期了。错误和数据输入时间的减少是显著的，人们对这个程序可能解决的其他问题充满了兴奋的讨论。即使主管也加入了头脑风暴，你强烈怀疑你很快就会被要求添加一些新功能。然而，有一个问题；这个应用程序已经是几百行的脚本了，你担心随着它的增长，它的可管理性。你需要花一些时间来组织你的代码库，为未来的扩展做准备。

在本章中，我们将学习以下主题：

+   如何使用**模型-视图-控制器**模式来分离应用程序的关注点

+   如何将代码组织成Python包

+   为您的包结构创建基本文件和目录

+   如何使用Git版本控制系统跟踪您的更改

# 分离关注点

适当的建筑设计对于任何需要扩展的项目都是至关重要的。任何人都可以支撑起一些支柱，建造一个花园棚屋，但是建造一座房子或摩天大楼需要仔细的规划和工程。软件也是一样的；简单的脚本可以通过一些快捷方式，比如全局变量或直接操作类属性来解决，但随着程序的增长，我们的代码需要以一种限制我们需要在任何给定时刻理解的复杂度的方式来隔离和封装不同的功能。

我们称之为**关注点的分离**，通过使用描述不同应用程序组件及其交互方式的架构模式来实现。

# MVC模式

这些模式中最持久的可能是MVC模式，它是在20世纪70年代引入的。尽管这种模式多年来已经发展并衍生出各种变体，但基本的要点仍然是：将数据、数据的呈现和应用程序逻辑保持在独立的组件中。

让我们更深入地了解这些组件，并在我们的应用程序的上下文中理解它们。

# 什么是模型？

MVC中的**模型**代表数据。这包括数据的存储，以及数据可以被查询或操作的各种方式。理想情况下，模型不关心或受到数据如何呈现或授予什么UI控件的影响，而是提供一个高级接口，只在最小程度上关注其他组件的内部工作。理论上，如果您决定完全更改程序的UI（比如，从Tkinter应用程序到Web应用程序），模型应该完全不受影响。

模型中包含的功能或信息的一些示例包括以下内容：

+   准备并将程序数据写入持久介质（数据文件、数据库等）

+   从文件或数据库中检索数据并将其转换为程序有用的格式

+   一组数据中字段的权威列表，以及它们的数据类型和限制

+   根据定义的数据类型和限制验证数据

+   对存储的数据进行计算

我们的应用程序目前没有模型类；数据布局是在表单类中定义的，到目前为止，`Application.on_save()`方法是唯一关心数据持久性的代码。我们需要将这个逻辑拆分成一个单独的对象，该对象将定义数据布局并处理所有CSV操作。

# 什么是视图？

**视图**是向用户呈现数据和控件的接口。应用程序可能有许多视图，通常是在相同的数据上。视图不直接与模型交互，并且理想情况下只包含足够的逻辑来呈现UI并将用户操作传递回控制器。

在视图中找到的一些代码示例包括以下内容：

+   GUI布局和小部件定义

+   表单自动化，例如字段的自动完成，小部件的动态切换，或错误对话框的显示

+   原始数据的格式化呈现

我们的`DataRecordForm`类是我们的主视图：它包含了我们应用程序用户界面的大部分代码。它还当前定义了我们数据记录的结构。这个逻辑可以留在视图中，因为视图确实需要一种在将数据临时传递给模型之前存储数据的方式，但从现在开始它不会再定义我们的数据记录。

随着我们继续前进，我们将向我们的应用程序添加更多视图。

# 什么是控制器？

**控制器**是应用程序的大中央车站。它处理用户的请求，并负责在视图和模型之间路由数据。MVC的大多数变体都会改变控制器的角色（有时甚至是名称），但重要的是它充当视图和模型之间的中介。我们的控制器对象将需要保存应用程序使用的视图和模型的引用，并负责管理它们之间的交互。

在控制器中找到的代码示例包括以下内容：

+   应用程序的启动和关闭逻辑

+   用户界面事件的回调

+   模型和视图实例的创建

我们的`Application`对象目前充当着应用程序的控制器，尽管它也包含一些视图和模型逻辑。随着应用程序的发展，我们将把更多的展示逻辑移到视图中，将更多的数据逻辑移到模型中，留下的主要是连接代码在我们的`Application`对象中。

# 为什么要复杂化我们的设计？

最初，以这种方式拆分应用程序似乎会增加很多不必要的开销。我们将不得不在不同对象之间传输数据，并最终编写更多的代码来完成完全相同的事情。为什么我们要这样做呢？

简而言之，我们这样做是为了使扩展可管理。随着应用程序的增长，复杂性也会增加。将我们的组件相互隔离限制了任何一个组件需要管理的复杂性的数量；例如，当我们重新构造表单视图的布局时，我们不应该担心模型将如何在输出文件中结构化数据。程序的这两个方面应该彼此独立。

这也有助于我们在放置某些类型的逻辑时保持一致。例如，拥有一个独立的模型对象有助于我们避免在UI代码中散布临时数据查询或文件访问尝试。

最重要的是，如果没有一些指导性的架构策略，我们的程序很可能会变成一团无法解开的逻辑混乱。即使不遵循严格的MVC设计定义，始终遵循松散的MVC模式也会在应用程序变得更加复杂时节省很多麻烦。

# 构建我们的应用程序目录结构

将程序逻辑上分解为单独的关注点有助于我们管理每个组件的逻辑复杂性，将代码物理上分解为多个文件有助于我们保持每个文件的复杂性可管理。这也加强了组件之间的隔离；例如，您不能共享全局变量，如果您的模型文件导入了`tkinter`，那么您就知道您做错了什么。

# 基本目录结构

Python应用程序目录布局没有官方标准，但有一些常见的约定可以帮助我们保持整洁，并且以后更容易打包我们的软件。让我们按照以下方式设置我们的目录结构：

1.  首先，创建一个名为`ABQ_Data_Entry`的目录。这是我们应用程序的**根目录**，所以每当我们提到**应用程序根目录**时，就是它。

1.  在应用程序根目录下，创建另一个名为`abq_data_entry`的目录。注意它是小写的。这将是一个Python包，其中将包含应用程序的所有代码；它应该始终被赋予一个相当独特的名称，以免与现有的Python包混淆。通常情况下，应用程序根目录和主模块之间不会有不同的大小写，但这也不会有任何问题；我们在这里这样做是为了避免混淆。

Python模块的命名应始终使用全部小写的名称和下划线。这个约定在PEP 8中有详细说明，PEP 8是Python的官方风格指南。有关PEP 8的更多信息，请参见[https://www.python.org/dev/peps/pep-0008](https://www.python.org/dev/peps/pep-0008)。

1.  接下来，在应用程序根目录下创建一个名为`docs`的文件夹。这个文件夹将用于存放关于应用程序的文档文件。

1.  最后，在应用程序根目录中创建两个空文件：`README.rst`和`abq_data_entry.py`。你的目录结构应该如下所示：

![](assets/830b3415-2492-4ad3-a86e-1e17b65c7a9b.png)

# abq_data_entry.py文件

就像以前一样，`abq_data_entry.py`是执行程序的主文件。不过，与以前不同的是，它不会包含大部分的程序。实际上，这个文件应该尽可能地简化。

打开文件并输入以下代码：

```py
from abq_data_entry.application import Application

app = Application()
app.mainloop()
```

保存并关闭文件。这个文件的唯一目的是导入我们的`Application`类，创建一个实例，并运行它。其余的工作将在`abq_data_entry`包内进行。我们还没有创建这个包，所以这个文件暂时无法运行；在我们处理文档之前，让我们先处理一下文档。

# README.rst文件

自上世纪70年代以来，程序一直包含一个名为`README`的简短文本文件，其中包含程序文档的简要摘要。对于小型程序，它可能是唯一的文档；对于大型程序，它通常包含用户或管理员的基本预先飞行指令。

`README`文件没有规定的内容集，但作为基本指南，考虑以下部分：

+   **描述**：程序及其功能的简要描述。我们可以重用规格说明中的描述，或类似的描述。这可能还包含主要功能的简要列表。

+   **作者信息**：作者的姓名和版权日期。如果你计划分享你的软件，这一点尤为重要，但即使对于公司内部的软件，让未来的维护者知道谁创建了软件以及何时创建也是有用的。

+   **要求**：软件和硬件要求的列表，如果有的话。

+   **安装**：安装软件、先决条件、依赖项和基本设置的说明。

+   **配置**：如何配置应用程序以及有哪些选项可用。这通常针对命令行或配置文件选项，而不是在程序中交互设置的选项。

+   **用法**：启动应用程序的描述，命令行参数和用户需要了解的其他注意事项。

+   **一般注意事项**：用户应该知道的注意事项或关键信息。

+   **错误**：应用程序中已知的错误或限制的列表。

并不是所有这些部分都适用于每个程序；例如，ABQ数据输入目前没有任何配置选项，所以没有理由有一个配置部分。根据情况，你可能会添加其他部分；例如，公开分发的软件可能会有一个常见问题解答部分，或者开源软件可能会有一个包含如何提交补丁的贡献部分。

`README`文件以纯ASCII或Unicode文本编写，可以是自由格式的，也可以使用标记语言。由于我们正在进行一个Python项目，我们将使用reStructuredText，这是Python文档的官方标记语言（这就是为什么我们的文件使用`rst`文件扩展名）。

# ReStructuredText

reStructuredText标记语言是Python `docutils`项目的一部分，完整的参考资料可以在Docutils网站找到：[http://docutils.sourceforge.net](http://docutils.sourceforge.net)。`docutils`项目还提供了将RST转换为PDF、ODT、HTML和LaTeX等格式的实用程序。

基础知识可以很快掌握，所以让我们来看看它们：

+   段落是通过在文本块之间留下一个空行来创建的。

+   标题通过用非字母数字符号下划线单行文本来创建。确切的符号并不重要；你首先使用的符号将被视为文档其余部分的一级标题，你其次使用的符号将被视为二级标题，依此类推。按照惯例，`=`通常用于一级，`-`用于二级，`~`用于三级，`+`用于四级。

+   标题和副标题的创建方式与标题相似，只是在上下都有一行符号。

+   项目列表是通过在行首加上`*`、`-`或`+`和一个空格来创建的。切换符号将创建子列表，多行点由将后续行缩进到文本从第一个项目符号开始的位置来创建。

+   编号列表的创建方式与项目列表相似，但使用数字（不需要正确排序）或`#`符号作为项目符号。

+   代码示例可以通过用双反引号字符括起来来指定内联(```py`), or in a block by ending a lead-in line with a double colon and indenting the code block.
*   Tables can either be created by surrounding columns of text with `=` symbols, separated by spaces to indicate the column breaks, or by constructing ASCII-art tables from `|`, `-`, and `+`. Tables can be tedious to create in a plain text editor, but some programming tools have plugins to generate the RST tables.

We've used RST already in [Chapter 2](3ec510a4-0919-4f25-9c34-f7bbd4199912.xhtml), *Designing GUI Applications with Tkinter,* to create our program specification; there, you saw the use of titles, headers, bullets, and a table. Let's walk through creating our `README.rst` file:

1.  Open the file and start with the title and description, as follows:

```

============================

ABQ数据输入应用程序

============================

描述

===========

此程序为ABQ Agrilabs实验室数据提供数据输入表单。

特点

--------

* 提供经过验证的输入表单，以确保正确的数据

* 将数据存储到ABQ格式的CSV文件中

* 在可能的情况下自动填充表单字段

```py

2.  Next, we'll list the authors by adding the following code:

```

作者

=======

Alan D Moore, 2018

```py

Add yourself, of course. Eventually, other people might work on your application; they should add their names here with the dates they worked on it. Now, add the requirements as follows:

```

要求

============

* Python 3

* Tkinter

```py

Right now, we only need Python 3 and Tkinter, but as our application grows we may be expanding this list. Our application doesn't really need to be installed, and has no configuration options, so for now we can skip those sections. Instead, we'll skip to `Usage` as follows:

```

用法

=====

要启动应用程序，请运行::

python3 ABQ_Data_Entry/abq_data_entry.py

```py

There really isn't much to know about running the program other than this command; no command-line switches or arguments. We don't know of any bugs, so we'll just leave some general notes at the end as follows:

```

一般说明

=============

CSV文件将以“abq_data_record_CURRENTDATE.csv”的格式保存在您当前的目录中，其中CURRENTDATE是今天的日期，采用ISO格式。

此程序仅追加到CSV文件。您应该安装电子表格程序，以防需要编辑或检查文件。

```py

It seems prudent to tell the user where the file will be saved and what it will be called, since that's hardcoded into the program right now. Also, we should mention the fact that the user should have some kind of spreadsheet, since the program can't edit or view the data. That finishes the `README.rst` file. Save it and let's move on to the `docs` folder.

# Populating the docs folder

The `docs` folder is where documentation goes. This can be any kind of documentation: user manuals, program specifications, API references, diagrams, and so on.

For now, you copy in the program specification we wrote in previous chapters, your interface mockups, and a copy of the form used by the technicians.

At some point, you might need to write a user manual, but for now the program is simple enough not to need it.

# Making a Python package

Creating your own Python package is surprisingly easy. A Python package consists of the following three things:

*   A directory
*   One or more Python files in that directory
*   A file called `__init__.py` in the directory

Once you've done this, you can import your package in whole or in part, just like you would import standard library packages, provided your script is in the same parent directory as the package directory.

Note that  `__init__.py` in a module is somewhat analogous to what `self.__init__()` is for a class. Code inside it will run whenever the package is imported. The Python community generally discourages putting much code in this file, though, and since no code is actually required, we'll leave this file empty.

Let's start building our application's package. Create the following six empty files under `abq_data_entry`:

*   `__init__.py`
*   `widgets.py`
*   `views.py`
*   `models.py`
*   `application.py`
*   `constants.py`

Each of those Python files is called a **module**. A module is nothing more than a Python file inside a package directory. Your directory structure should now look like this:

![](assets/06efc903-784c-426e-be9b-ddeb66de7849.png)

At this point, you have a working package, albeit with no actual code in it. To test this, open a Terminal/command-line window, change to your `ABQ_Data_Entry` directory, and start a Python shell.

Now, type the following command:

```

from abq_data_entry import application

```py

This should work without error. Of course, it doesn't do anything, but we'll get to that next.

Don't confuse the term package here with the actual distributable Python packages, such as those you download using `pip`. 

# Splitting our application into multiple files

Now that our directory structure is in order, we need to start dissecting our application script and splitting it up into our module files. We'll also need to create our model class. Open up your `abq_data_entry.py` file from [Chapter 4](43851b46-13ed-4f6f-b754-bc9fe4f522d9.xhtml), *Reducing User Error with Validation and Automation,* and let's begin!

# Creating the models module

When your application is all about data, it's good to begin with the model. Remember that the job of a model is to manage the storage, retrieval, and processing of our application's data, usually with respect to its persistent storage format (in this case, CSV). To accomplish this, our model should contain all the knowledge about our data.

Currently, our application has nothing like a model; knowledge about the application's data is scattered into the form fields, and the `Application` object simply takes whatever data the form contains and stuffs it directly into a CSV file when a save operation is requested. Since we aren't yet retrieving or updating information, our application has no actual knowledge about what's inside the CSV file.

To move our application to an MVC architecture, we'll need to create a model class that both manages data storage and retrieval, and represents the authoritative source of knowledge about our data. In other words, we have to encode the knowledge contained in our data dictionary here in our model. We don't really know what we'll *do* with this knowledge yet, but this is where it belongs.

There are a few ways we could store this data, such as creating a custom field class or a `namedtuple` object, but we'll keep it simple for now and just use a dictionary, mapping field names to field metadata.

The field metadata will likewise be stored as a dictionary of attributes about the field, which will include: 

*   Whether or not the field is required
*   The type of data stored in the field
*   The list of possible values, if applicable
*   The minimum, maximum, and increment of values, if applicable

To store the data type for each field, let's define some data types. Open the `constants.py` file and add the following code:

```

class FieldTypes:

string = 1

string_list = 2

iso_date_string = 3

long_string = 4

decimal = 5

integer = 6

boolean = 7

```py

We've created a class called `FieldTypes` that simply stores some named integer values, which will describe the different types of data we're going to store. We could just use Python types here, but it's useful to differentiate between certain types of data that are likely to be the same Python type (such as `long`, `short`, and `date` strings). Note that the integer values here are basically meaningless; they just need to be different from one another.

Python 3 has an `Enum` class, which we could have used here, but it adds very little that we actually need in this case. You may want to investigate this class if you're creating a lot of constants such as our `FieldTypes` class and need additional features.

Now, open `models.py`, where we'll import `FieldTypes` and create our model class and field definitions as follows:

```

import csv

import os

from .constants import FieldTypes as FT

class CSVModel:

"""CSV文件存储"""

字段 = {

"日期": {'req': True, 'type': FT.iso_date_string},

"时间": {'req': True, 'type': FT.string_list,

'values': ['8:00', '12:00', '16:00', '20:00']},

"技术员": {'req': True, 'type':  FT.string},

"实验室": {'req': True, 'type': FT.string_list,

'values': ['A', 'B', 'C', 'D', 'E']},

"情节": {'req': True, 'type': FT.string_list,

'values': [str(x) for x in range(1, 21)]},

"种子样本":  {'req': True, 'type': FT.string},

"湿度": {'req': True, 'type': FT.decimal,

'min': 0.5, 'max': 52.0, 'inc': .01},

"光": {'req': True, 'type': FT.decimal,

'min': 0, 'max': 100.0, 'inc': .01},

"温度": {'req': True, 'type': FT.decimal,

'min': 4, 'max': 40, 'inc': .01},

"设备故障": {'req': False, 'type': FT.boolean},

"植物": {'req': True, 'type': FT.integer,

'min': 0, 'max': 20},

"花": {'req': True, 'type': FT.integer,

'min': 0, 'max': 1000},

"水果": {'req': True, 'type': FT.integer,

'min': 0, 'max': 1000},

"最小高度": {'req': True, 'type': FT.decimal,

'min': 0, 'max': 1000, 'inc': .01},

"最大高度": {'req': True, 'type': FT.decimal,

'min': 0, 'max': 1000, 'inc': .01},

"中位数高度": {'req': True, 'type': FT.decimal,

'min': 0, 'max': 1000, 'inc': .01},

"注释": {'req': False, 'type': FT.long_string}

}

```py

Notice the way we import `FieldTypes`:  `from .constants import FieldTypes`. The dot in front of `constants` makes this a **relative import**. Relative imports can be used inside a Python package to locate other modules in the same package. In this case, we're in the `models` module, and we need to access the `constants` module inside the `abq_data_entry` package. The single dot represents our current parent module (`abq_data_entry`), and thus `.constants` means the `constants` module of the `abq_data_entry` package.

Relative imports also distinguish our custom modules from modules in `PYTHONPATH`. Thus, we don't have to worry about any third-party or standard library packages conflicting with our module names.

In addition to field attributes, we're also documenting the order of fields here. In Python 3.6 and later, dictionaries retain the order they were defined by; if you're using an older version of Python 3, you'd need to use the `OrderedDict` class from the `collections` standard library module to preserve the field order.

Now that we have a class that understands which fields need to be stored, we need to migrate our save logic from the application class into the model.

The code in our current script is as follows:

```

datestring = datetime.today().strftime("%Y-%m-%d")

filename = "abq_data_record_{}.csv".format(datestring)

newfile = not os.path.exists(filename)

data = self.recordform.get()

with open(filename, 'a') as fh:

csvwriter = csv.DictWriter(fh, fieldnames=data.keys())

if newfile:

csvwriter.writeheader()

csvwriter.writerow(data)

```py

Let's go through this code and determine what goes to the model and what stays in the controller (that is, the `Application` class):

*   The first two lines define the filename we're going to use. This could go into the model, but thinking ahead, it seems that the users may want to be able to open arbitrary files or define the filename manually. This means the application will need to be able to tell the model which filename to work with, so it's better to leave the logic that determines the name in the controller.
*   The `newfile` line determines whether the file exists or not. As an implementation detail of the data storage medium, this is clearly the model's problem, not the application's.
*   `data = self.recordform.get()` pulls data from the form. Since our model has no knowledge of the form's existence, this needs to stay in the controller.
*   The last block opens the file, creates a `csv.DictWriter` object, and appends the data. This is definitely the model's concern.

Now, let's begin moving code into the `CSVModel` class:

1.  To start the process, let's create a constructor for `CSVModel` that allows us to pass in a filename:

```

def __init__(self, filename):

self.filename = filename

```py

The constructor is pretty simple; it just takes a `filename` parameter and stores it as a property. Now, we'll migrate the save logic as follows:

```

def save_record(self, data):

"""将数据字典保存到CSV文件"""

newfile = not os.path.exists(self.filename)

with open(self.filename, 'a') as fh:

csvwriter = csv.DictWriter(fh,

fieldnames=self.fields.keys())

if newfile:

csvwriter.writeheader()

csvwriter.writerow(data)

```py

This is essentially the logic we chose to copy from `Application.on_save()`, but with one difference; in the call to `csv.DictWriter()`, the `fieldnames` parameter is defined by the model's `fields` list rather than the keys of the `data` dict. This allows our model to manage the format of the CSV file itself, and not depend on what the form gives it.

2.  Before we're done, we need to take care of our module imports. The `save_record()` method uses the `os` and `csv` libraries, so we need to import them. Add this to the top of the file as follows:

```

import csv

import os

```py

With the model in place, let's start working on our view components.

# Moving the widgets

While we could put all of our UI-related code in one `views` file, we have a lot of widget classes that should really be put in their own file to limit the complexity of the `views` file.

So instead, we're going to move all of the code for our widget classes into the `widgets.py` file. Widgets include all the classes that implement reusable GUI components, including compound widgets like `LabelInput`. As we develop more of these, we'll add them to this file.

Open `widgets.py` and copy in all of the code for `ValidatedMixin`, `DateInput`, `RequiredEntry`, `ValidatedCombobox`, `ValidatedSpinbox`, and `LabelInput`. These are our widgets.

The `widgets.py` file will need to import any module dependencies used by the code being copied in. We'll need to look through our code and find what libraries we use and import them. Obviously, we need `tkinter` and `ttk`, so add those at the top as follows:

```

import tkinter as tk

from tkinter import ttk

```py

Our `DateInput` class uses the `datetime` class from the `datetime` library, so import that too, as follows:

```

from datetime import datetime

```py

Finally, our `ValidatedSpinbox` class makes use of the `Decimal` class and `InvalidOperation` exception from the `decimal` library as follows:

```

from decimal import Decimal, InvalidOperation

```py

This is all we need in `widgets.py` for now, but we'll revisit this file as we refactor our view logic.

# Moving the views

Next, we need to create the `views.py` file. Views are larger GUI components, like our `DataRecordForm` class. Currently it's our only view, but we'll be creating more views in later chapters, and they'll be added here.

Open the `views.py` file and copy in the `DataRecordForm` class, then go back to the top to deal with the module imports. Again, we'll need `tkinter` and `ttk`, and our file saving logic relies on `datetime` for the filename.

Add them to the top of the file as follows:

```

import tkinter as tk

from tkinter import ttk

from datetime import datetime

```py

We aren't done, though; our actual widgets aren't here and we'll need to import them. Since we're going to be doing a lot of importing of objects between our files, let's pause for a moment to consider the best way to handle these imports.

There are three ways we could import objects:

*   Use a wildcard import to bring in all the classes from `widgets.py`
*   Explicitly import all the needed classes from `widgets.py` using the `from ... import ...` format
*   Import `widgets` and keep our widgets in their own namespace

Let's consider the relative merits of those ways:

*   The first option is by far the easiest, but it can cause us headaches as the application expands. A wildcard import will bring in every name defined at the global scope within the module. That includes not just the classes we defined, but any imported modules, aliases, and defined variables or functions. This can lead to unintended consequences and subtle bugs as the application expands in complexity.
*   The second option is cleaner, but means we'll need to maintain the list of imports as we add new classes and use them in different files, and this leads to a long and ugly imports section that is hard for humans to parse.
*   The third option is by far the best, as it keeps all names within a namespace and keeps the code elegantly simple. The only downside is that we'll need to update our code so that all references to widget classes include the module name as well. To keep this from being unwieldy, let's alias the `widgets` module to something short, like `w`.

Add the following code to your imports:

```

from . import widgets as w

```py

Now, we just need to go through the code and prepend `w.` to all instances of `LabelInput`, `RequiredEntry`, `DateEntry`, `ValidatedCombobox`, and `ValidatedSpinbox`. This should be easy enough to do in IDLE or any other text editor using a series of search and replace actions.

For example, `line 1` of the form is as follows:

```

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

```py

Before you go through and change that everywhere, though, let's stop and take a moment to refactor some of the redundancy out of this code.

# Removing redundancy in our view logic

Look at the field definitions in the view logic: they contain a lot of information that is also in our model. Minimums, maximums, increments, and possible values are defined both here and in our model code. Even the type of the input widget is related directly to the type of data being stored. Ideally, this should only be defined one place, and that place should be the model. If we needed to update the model for some reason, our form would be out of sync.

What we need to do is to pass the field specification from our model into the view class and let the widgets' details be defined from that specification.

Since our widget instances are being defined inside the `LabelInput` class, we're going to enhance that class with the ability to automatically work out the `input` class and arguments from our model's field specification format. Open up the `widgets.py` file and import the `FieldTypes` class, just as you did in `model.py`. 

Now, locate the `LabelInput` class and add the following code before the `__init__()` method:

```

field_types = {

FT.string: (RequiredEntry, tk.StringVar),

FT.string_list: (ValidatedCombobox, tk.StringVar),

FT.iso_date_string: (DateEntry, tk.StringVar),

FT.long_string: (tk.Text, lambda: None),

FT.decimal: (ValidatedSpinbox, tk.DoubleVar),

```py

```

FT.integer: (ValidatedSpinbox, tk.IntVar),

FT.boolean: (ttk.Checkbutton, tk.BooleanVar)

}

```py

This code acts as a key to translate our model's field types into a widget type and variable type appropriate for the field type.

Now, we need to update `__init__()` to take a `field_spec` parameter and, if given, use it to define the input widget as follows:

```

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

如果字段规范中包含'min'，并且输入参数中不包含'from_'：

input_args['from_'] = field_spec.get('min')

如果字段规范中包含'max'，并且输入参数中不包含'to'：

input_args['to'] = field_spec.get('max')

if 'inc' in field_spec and 'increment' not in input_args:

input_args['increment'] = field_spec.get('inc')

# values

如果字段规范中包含'values'，并且输入参数中不包含'values'：

input_args['values'] = field_spec.get('values')

else:

self.variable = input_var        if input_class in (ttk.Checkbutton, ttk.Button, ttk.Radiobutton):

input_args["text"] = label

input_args["variable"] = self.variable

else:

self.label = ttk.Label(self, text=label, **label_args)

self.label.grid(row=0, column=0, sticky=(tk.W + tk.E))

input_args["textvariable"] = self.variable

# ... __init__()的其余部分相同

```py

Let's break down the changes:

1.  First, we've added `field_spec` as a keyword argument with `None` as a default. We might want to use this class in a situation where there isn't a field specification, so we keep this parameter optional.
2.  If there is `field_spec` given, we're going to do the following things:
    *   We'll grab the `type` value and use that with our class's field key to get `input_class`. In case we want to override this, an explicitly passed `input_class` will override the detected one.
    *   We'll determine the appropriate variable type in the same way. Once again, if `input_var` is explicitly passed, we'll prefer that, otherwise we'll use the one determined from the field type. We'll create an instance either way and store it in `self.variable`.
    *   For `min`, `max`, `inc`, and `values`, if the key exists in the field specification, and the corresponding `from_`, `to`, `increment`, or `values` argument has not been passed in explicitly, we'll set up the `input_args` variable with the `field_spec` value.
3.  If `field_spec` wasn't passed in, we need to assign `self.variable` from the `input_var` argument.
4.  We're using `self.variable` now instead of `input_var` for assigning the input's variable, since those values might not necessarily be the same anymore and `self.variable` will contain the correct reference.

Now, we can update our view code to take advantage of this new ability. Our `DataRecordForm` class will need access to the model's `fields` dictionary, which it can then use to send a field specification to the `LabelInput` class.

Back in the `views.py` file, edit the method signature so that we can pass in a dictionary of field specifications:

```

def __init__(self, parent, fields, *args, **kwargs):

```py

With access to the `fields` dictionary, we can just get the field specification from it and pass that into the `LabelInput` class instead of specifying the input class, input variable, and input arguments.

Now, the first line looks like this:

```

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

```py

Go ahead and update the rest of the widgets the same way, replacing `input_class`, `input_var`, and `input_args` with `field_spec`. Note that when you get to the height fields, you'll need to keep the part of `input_args` that defines `min_var`, `max_var`, and `focus_update_var`.

For example, the following is the `Min Height` input definition:

```

self.inputs['Min Height'] = w.LabelInput(

plantinfo, "Min Height (cm)",

field_spec=fields['Min Height'],

input_args={"max_var": max_height_var,

"focus_update_var": min_height_var})

```py

That does it. Now, any changes to our field specification can be made solely in the model, and the form will simply do the correct thing.

# Creating the application file

Finally, let's create our controller class, `Application`, by following these steps:

1.  Open the `application.py` file and copy in the `Application` class definition from the script.
2.  The first thing we'll fix is our imports. At the top of the file, add the following code:

```

import tkinter as tk

from tkinter import ttk

from datetime import datetime

from . import views as v

from . import models as m

```py

We need `tkinter` and `ttk`, of course, and `datetime` to define our filename. Although we only need one class each from `views` and `models`, we're going to keep them in their own namespaces anyway. It's likely we're going to have many more views as the application expands, and possibly more models.

3.  We need to update the call to `DataRecordForm` in `__init__()` for the new namespace and make sure we pass in the required field specification dictionary as follows:

```

self.recordform = v.DataRecordForm(self, m.CSVModel.fields)

```py

4.  Finally, we need to update `Application.on_save()` to use the model, as follows:

```

def on_save(self):

"""处理保存按钮点击"""

errors = self.recordform.get_errors()

if errors:

self.status.set(

"无法保存，字段错误：{}"

.format(', '.join(errors.keys())))

return False

# 目前，我们保存到一个硬编码的文件名

带有日期字符串。

datestring = datetime.today().strftime("%Y-%m-%d")

filename = "abq_data_record_{}.csv".format(datestring)

model = m.CSVModel(filename)

data = self.recordform.get()

模型保存记录（数据）

self.records_saved += 1

self.status.set(

"{}条记录已保存此会话"。

format(self.records_saved)

）

self.recordform.reset()

```py

As you can see, using our model is pretty seamless; we just created a `CSVModel` class by passing in the filename, and then passed the form's data to `save_record()`.

# Running the application

The application is now completely migrated to the new data format. To test it, navigate to the application root folder, `ABQ_Data_Entry`, and execute the following command:

```

python3 abq_data_entry.py

```py

It should look and act just like the single script from [Chapter 4](43851b46-13ed-4f6f-b754-bc9fe4f522d9.xhtml), *Reducing User Error with Validation and Automation,* and run without errors, as shown in the following screenshot:

![](assets/4151fc4d-d11b-4bf1-a5a3-df5ab3971dca.png)

Success!

# Using version control software

Our code is nicely structured for expansion, but there's one more critical item we should address: **version control**. You may already be familiar with a **version control system** (**VCS**), sometimes called **revision control** or **source code management**, but if not, it's an indispensable tool for dealing with a large and changing codebase.

When working on an application, we sometimes think we know what needs to be changed, but it turns out we're wrong. Sometimes we don't know exactly how to code something, and it takes several attempts to find the correct approach. Sometimes we need to revert to code that was changed a long time back. Sometimes we have multiple people working on the same piece of code, and we need to merge their changes together. Version control systems were created to address these issues and more.

There are dozens of different version control systems, but most of them work essentially the same:

*   You have a working copy of the code that you make changes to
*   You periodically select changes to commit back to the master copy
*   You can checkout older versions of the code at any point, then revert back to the master copy
*   You can create branches of the code to experiment with different approaches, new features, or large refactors
*   You can later merge these branches back into the master copy

VCS provides a safety net that gives you the freedom to change your code without the fear that you'll hopelessly ruin it: reverting to a known working state is just a few quick commands away. It also helps us to document changes to our code, and collaborate with others if the opportunity arises.

There are dozens of VC systems available, but by far the most popular for many years now is **Git**.

# A super-quick guide to using Git

Git was created by Linus Torvalds to be the version control software for the Linux kernel project, and has since grown to be the most popular VC software in the world. It is utilized by source sharing sites like GitHub, Bitbucket, SourceForge, and GitLab. Git is extremely powerful, and mastering it can take months or years; fortunately, the basics can be grasped in a few minutes.

First, you'll need to install Git; visit [https://git-scm.com/downloads](https://git-scm.com/downloads) for instructions on how to install Git on macOS, Windows, Linux, or other Unix operating systems.

# Initializing and configuring a Git repository

Once Git is installed, we need to initialize and configure our project directory as a Git repository by following these steps:

1.  Run the following command in the application's root directory (`ABQ_Data_Entry`):

```

git init

```py

This command creates a hidden directory under our project root called `.git` and initializes it with the basic files that make up the repository. The `.git` directory will contain all the data and metadata about our saved revisions.

2.  Before we add any files to the repository, we need to instruct Git to ignore certain kinds of files. For example, Python creates bytecode (`.pyc`) files whenever it executes a file, and we don't want to save these as part of our code. To do this, create a file in your project root called `.gitignore` and put the following lines in it:

```

*.pyc

__pycache__/

```py

# Adding and committing code

Now that our repository is initialized, we can add files and directories to our Git repository using the following commands:

```

git add abq_data_entry

git add abq_data_entry.py

git add docs

git add README.rst

```py

At this point, our files are staged, but not yet committed to the repository. You can check the status of your repository and the files in it at any time by entering `git status`.

You should get the following output:

```

在主分支上

还没有提交

要提交的更改：

（使用“git rm --cached <file>…”取消暂存）

新文件：README.rst

新文件：abq_data_entry.py

新文件：abq_data_entry/__init__.py

新文件：abq_data_entry/application.py

新文件：abq_data_entry/models.py

新文件：abq_data_entry/views.py

新文件：abq_data_entry/widgets.py

新文件：docs/Application_layout.png

新文件：docs/abq_data_entry_spec.rst

新文件：docs/lab-tech-paper-form.png

未跟踪的文件：

（使用“git add <file>…”将其包含在将要提交的内容中）

.gitignore

```py

This shows you that all the files under `abq_data_entry` and `docs`, as well as the files you specified directly, are staged to be committed to the repository.

Let's go ahead and commit the changes as follows:

```

git commit -m "Initial commit"

```py

The `-m` flag here passes in a commit message, which is stored with the commit. Each time you commit code to the repository, you will be required to write a message. You should always make these messages as meaningful as possible, detailing what changes you made and the rationale behind them.

# Viewing and using our commits

To view your repository's history, run the `git log` command as follows:

```

alanm@alanm-laptop:~/ABQ_Data_Entry$ git log

提交df48707422875ff545dc30f4395f82ad2d25f103（HEAD -> master）

作者：Alan Moore <alan@example.com>

日期：2017年12月21日星期四18:12:17 -0600

初始提交

```

正如您所看到的，我们上次提交的`作者`、`日期`和`提交`消息都显示出来。如果我们有更多的提交，它们也会在这里列出，从最新到最旧。您在输出的第一行中看到的长十六进制值是**提交哈希**，这是一个唯一的值，用于标识提交。这个值可以用来在其他操作中引用提交。

例如，我们可以使用它将我们的存储库重置到过去的状态，如下所示：

1.  删除`README.rst`文件，并验证它已完全消失。

1.  现在，输入命令`git reset --hard df48707`，将`df48707`替换为您提交哈希的前七个字符。

1.  再次检查您的文件清单：`README.rst`文件已经回来了。

这里发生的是我们改变了我们的存储库，然后告诉Git将存储库的状态硬重置到我们的第一个提交。如果您不想重置您的存储库，您也可以暂时检出一个旧的提交，或者使用特定的提交作为基础创建一个分支。正如您已经看到的，这为我们提供了一个强大的实验安全网；无论您如何调整代码，任何提交都只是一个命令的距离！

Git有许多更多的功能超出了本书的范围。如果您想了解更多信息，Git项目在[https://git-scm.com/book](https://git-scm.com/book)提供了免费的在线手册，您可以在那里了解分支和设置远程存储库等高级功能。目前，重要的是在进行更改时提交更改，以便保持您的安全网并记录更改的历史。

# 总结

在本章中，您学会了为您的简单脚本做一些严肃的扩展准备。您学会了如何将应用程序的职责领域划分为单独的组件，以及如何将代码分割成单独的模块。您学会了如何使用reStructuredText记录代码并使用版本控制跟踪所有更改。

在下一章中，我们将通过实现一些新功能来测试我们的新项目布局。您将学习如何使用Tkinter的应用程序菜单小部件，如何实现文件打开和保存，以及如何使用消息弹出窗口来警告用户或确认操作。
