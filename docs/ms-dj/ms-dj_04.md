# 第四章：模型

在第二章*视图和 URLconfs*中，我们介绍了使用 Django 构建动态网站的基础知识：设置视图和 URLconfs。正如我们所解释的，视图负责执行一些任意逻辑，然后返回一个响应。在其中一个示例中，我们的任意逻辑是计算当前日期和时间。

在现代 Web 应用程序中，任意逻辑通常涉及与数据库的交互。在幕后，一个数据库驱动的网站连接到数据库服务器，从中检索一些数据，并在网页上显示这些数据。该网站还可能提供访问者自行填充数据库的方式。

许多复杂的网站提供了这两种方式的组合。例如，[www.amazon.com](http://www.amazon.com)就是一个数据库驱动的网站的绝佳例子。每个产品页面本质上都是对亚马逊产品数据库的查询，格式化为 HTML，当您发布客户评论时，它会被插入到评论数据库中。

Django 非常适合制作数据库驱动的网站，因为它提供了使用 Python 执行数据库查询的简单而强大的工具。本章解释了这个功能：Django 的数据库层。

### 注意

虽然不是必须要了解基本的关系数据库理论和 SQL 才能使用 Django 的数据库层，但强烈建议这样做。这本书不涉及这些概念的介绍，但即使你是数据库新手，继续阅读也是有可能跟上并理解基于上下文的概念。

# 在视图中进行数据库查询的“愚蠢”方法

正如第二章*视图和 URLconfs*中详细介绍了在视图中生成输出的“愚蠢”方法（通过在视图中直接硬编码文本），在视图中从数据库中检索数据也有一个“愚蠢”的方法。很简单：只需使用任何现有的 Python 库来执行 SQL 查询并对结果进行处理。在这个示例视图中，我们使用`MySQLdb`库连接到 MySQL 数据库，检索一些记录，并将它们传递给模板以在网页上显示：

```py
from django.shortcuts import render 
import MySQLdb 

def book_list(request): 
    db = MySQLdb.connect(user='me', db='mydb',  passwd='secret', host='localhost') 
    cursor = db.cursor() 
    cursor.execute('SELECT name FROM books ORDER BY name') 
    names = [row[0] for row in cursor.fetchall()] 
    db.close() 
    return render(request, 'book_list.html', {'names': names}) 

```

这种方法可以工作，但是一些问题应该立即引起您的注意：

+   我们在硬编码数据库连接参数。理想情况下，这些参数应该存储在 Django 配置中。

+   我们不得不写相当多的样板代码：创建连接，创建游标，执行语句，关闭连接。理想情况下，我们只需要指定我们想要的结果。

+   它将我们与 MySQL 绑定。如果将来我们从 MySQL 切换到 PostgreSQL，我们很可能需要重写大量代码。理想情况下，我们使用的数据库服务器应该被抽象化，这样数据库服务器的更改可以在一个地方进行。 （如果您正在构建一个希望尽可能多的人使用的开源 Django 应用程序，这个功能尤其重要。）

正如您所期望的，Django 的数据库层解决了这些问题。

# 配置数据库

考虑到所有这些理念，让我们开始探索 Django 的数据库层。首先，让我们探索在创建应用程序时添加到`settings.py`的初始配置。

```py
# Database 
#  
DATABASES = { 
    'default': { 
        'ENGINE': 'django.db.backends.sqlite3', 
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'), 
    } 
} 

```

默认设置非常简单。以下是每个设置的概述。

+   `ENGINE`：它告诉 Django 使用哪个数据库引擎。在本书的示例中，我们使用 SQLite，所以将其保留为默认的`django.db.backends.sqlite3`。

+   `NAME`：它告诉 Django 你的数据库的名称。例如：`'NAME': 'mydb',`。

由于我们使用的是 SQLite，`startproject`为我们创建了数据库文件的完整文件系统路径。

这就是默认设置-你不需要改变任何东西来运行本书中的代码，我包含这个只是为了让你了解在 Django 中配置数据库是多么简单。有关如何设置 Django 支持的各种数据库的详细描述，请参见第二十一章, *高级数据库管理*。

# 你的第一个应用程序

现在你已经验证了连接是否正常工作，是时候创建一个**Django 应用程序**了-一个包含模型和视图的 Django 代码包，它们一起存在于一个单独的 Python 包中，代表一个完整的 Django 应用程序。这里值得解释一下术语，因为这往往会让初学者困惑。我们已经在第一章中创建了一个项目，*Django 简介和入门*，那么**项目**和**应用程序**之间有什么区别呢？区别在于配置和代码：

+   项目是一组 Django 应用程序的实例，以及这些应用程序的配置。从技术上讲，项目的唯一要求是提供一个设置文件，其中定义了数据库连接信息、已安装应用程序的列表、`DIRS`等。

+   应用程序是一组可移植的 Django 功能，通常包括模型和视图，它们一起存在于一个单独的 Python 包中。

例如，Django 自带了许多应用程序，比如自动管理界面。关于这些应用程序的一个关键点是它们是可移植的，可以在多个项目中重复使用。

关于如何将 Django 代码适应这个方案，几乎没有硬性规定。如果你正在构建一个简单的网站，可能只使用一个应用程序。如果你正在构建一个包括电子商务系统和留言板等多个不相关部分的复杂网站，你可能希望将它们拆分成单独的应用程序，以便将来可以单独重用它们。

事实上，你并不一定需要创建应用程序，正如我们在本书中迄今为止创建的示例视图函数所证明的那样。在这些情况下，我们只需创建一个名为`views.py`的文件，填充它以视图函数，并将我们的 URLconf 指向这些函数。不需要应用程序。

然而，关于应用程序约定有一个要求：如果你正在使用 Django 的数据库层（模型），你必须创建一个 Django 应用程序。模型必须存在于应用程序中。因此，为了开始编写我们的模型，我们需要创建一个新的应用程序。

在`mysite`项目目录中（这是你的`manage.py`文件所在的目录，而不是`mysite`应用程序目录），输入以下命令来创建一个`books`应用程序：

```py
python manage.py startapp books

```

这个命令不会产生任何输出，但它会在`mysite`目录中创建一个`books`目录。让我们看看该目录的内容：

```py
books/ 
    /migrations 
    __init__.py 
    admin.py 
    models.py 
    tests.py 
    views.py 

```

这些文件将包含此应用程序的模型和视图。在你喜欢的文本编辑器中查看`models.py`和`views.py`。这两个文件都是空的，除了注释和`models.py`中的导入。这是你的 Django 应用程序的空白板。

# 在 Python 中定义模型

正如我们在第一章中讨论的那样，MTV 中的 M 代表模型。Django 模型是对数据库中数据的描述，表示为 Python 代码。它是你的数据布局-相当于你的 SQL `CREATE TABLE`语句-只不过它是用 Python 而不是 SQL 编写的，并且包括的不仅仅是数据库列定义。

Django 使用模型在后台执行 SQL 代码，并返回表示数据库表中行的方便的 Python 数据结构。Django 还使用模型来表示 SQL 不能必然处理的更高级概念。

如果你熟悉数据库，你可能会立刻想到，“在 Python 中定义数据模型而不是在 SQL 中定义，这不是多余的吗？” Django 之所以采用这种方式有几个原因：

+   内省需要额外开销，而且并不完美。为了提供方便的数据访问 API，Django 需要以某种方式了解数据库布局，有两种方法可以实现这一点。第一种方法是在 Python 中明确描述数据，第二种方法是在运行时内省数据库以确定数据模型。

+   这第二种方法看起来更干净，因为关于你的表的元数据只存在一个地方，但它引入了一些问题。首先，在运行时内省数据库显然需要开销。如果框架每次处理请求时，甚至只在 Web 服务器初始化时都需要内省数据库，这将产生无法接受的开销。（虽然有些人认为这种开销是可以接受的，但 Django 的开发人员的目标是尽量减少框架的开销。）其次，一些数据库，特别是较旧版本的 MySQL，没有存储足够的元数据来进行准确和完整的内省。

+   编写 Python 很有趣，而且将所有东西都放在 Python 中可以减少你的大脑进行“上下文切换”的次数。如果你尽可能长时间地保持在一个编程环境/思维方式中，这有助于提高生产率。不得不先写 SQL，然后写 Python，再写 SQL 是会打断思维的。

+   将数据模型存储为代码而不是在数据库中，可以更容易地将模型纳入版本控制。这样，你可以轻松跟踪对数据布局的更改。

+   SQL 只允许对数据布局进行一定级别的元数据。例如，大多数数据库系统并没有提供专门的数据类型来表示电子邮件地址或 URL。但 Django 模型有。更高级别的数据类型的优势在于更高的生产率和更可重用的代码。

+   SQL 在不同的数据库平台上是不一致的。例如，如果你要分发一个网络应用程序，更实际的做法是分发一个描述数据布局的 Python 模块，而不是针对 MySQL、PostgreSQL 和 SQLite 分别创建`CREATE TABLE`语句的集合。

然而，这种方法的一个缺点是，Python 代码可能与实际数据库中的内容不同步。如果你对 Django 模型进行更改，你需要在数据库内做相同的更改，以保持数据库与模型一致。在本章后面讨论迁移时，我将向你展示如何处理这个问题。

最后，你应该注意到 Django 包括一个实用程序，可以通过内省现有数据库来生成模型。这对于快速启动和运行遗留数据非常有用。我们将在第二十一章中介绍这个内容，*高级数据库管理*。

## 你的第一个模型

作为本章和下一章的一个持续的例子，我将专注于一个基本的书籍/作者/出版商数据布局。我选择这个作为例子，因为书籍、作者和出版商之间的概念关系是众所周知的，这是初级 SQL 教科书中常用的数据布局。你也正在阅读一本由作者撰写并由出版商出版的书籍！

我假设以下概念、字段和关系：

+   作者有名字、姓氏和电子邮件地址。

+   出版商有一个名称、街道地址、城市、州/省、国家和网站。

+   一本书有一个标题和出版日期。它还有一个或多个作者（与作者之间是多对多的关系）和一个出版商（一对多的关系，也就是外键到出版商）。

在 Django 中使用这个数据库布局的第一步是将其表达为 Python 代码。在由`startapp`命令创建的`models.py`文件中输入以下内容：

```py
from django.db import models 

class Publisher(models.Model): 
    name = models.CharField(max_length=30) 
    address = models.CharField(max_length=50) 
    city = models.CharField(max_length=60) 
    state_province = models.CharField(max_length=30) 
    country = models.CharField(max_length=50) 
    website = models.URLField() 

class Author(models.Model): 
    first_name = models.CharField(max_length=30) 
    last_name = models.CharField(max_length=40) 
    email = models.EmailField() 

class Book(models.Model): 
    title = models.CharField(max_length=100) 
    authors = models.ManyToManyField(Author) 
    publisher = models.ForeignKey(Publisher) 
    publication_date = models.DateField() 

```

让我们快速检查这段代码，以涵盖基础知识。首先要注意的是，每个模型都由一个 Python 类表示，该类是`django.db.models.Model`的子类。父类`Model`包含使这些对象能够与数据库交互所需的所有机制，这样我们的模型就只负责以一种简洁而紧凑的语法定义它们的字段。

信不信由你，这就是我们需要编写的所有代码，就可以使用 Django 进行基本的数据访问。每个模型通常对应一个单独的数据库表，模型上的每个属性通常对应该数据库表中的一列。属性名称对应于列的名称，字段类型（例如，`CharField`）对应于数据库列类型（例如，`varchar`）。例如，`Publisher`模型等效于以下表（假设使用 PostgreSQL 的`CREATE TABLE`语法）：

```py
CREATE TABLE "books_publisher" ( 
    "id" serial NOT NULL PRIMARY KEY, 
    "name" varchar(30) NOT NULL, 
    "address" varchar(50) NOT NULL, 
    "city" varchar(60) NOT NULL, 
    "state_province" varchar(30) NOT NULL, 
    "country" varchar(50) NOT NULL, 
    "website" varchar(200) NOT NULL 
); 

```

事实上，Django 可以自动生成`CREATE TABLE`语句，我们将在下一刻向您展示。一个类对应一个数据库表的唯一规则的例外是多对多关系的情况。在我们的示例模型中，`Book`有一个名为`authors`的`ManyToManyField`。这表示一本书有一个或多个作者，但`Book`数据库表不会得到一个`authors`列。相反，Django 会创建一个额外的表-一个多对多的*连接表*-来处理书籍到作者的映射。

对于字段类型和模型语法选项的完整列表，请参见附录 B, *数据库 API 参考*。最后，请注意，我们没有在任何这些模型中明确定义主键。除非您另有指示，否则 Django 会自动为每个模型提供一个自增的整数主键字段，称为`id`。每个 Django 模型都需要有一个单列主键。

## 安装模型

我们已经编写了代码；现在让我们在数据库中创建表。为了做到这一点，第一步是在我们的 Django 项目中激活这些模型。我们通过将`books`应用程序添加到设置文件中的已安装应用程序列表中来实现这一点。再次编辑`settings.py`文件，并查找`INSTALLED_APPS`设置。`INSTALLED_APPS`告诉 Django 为给定项目激活了哪些应用程序。默认情况下，它看起来像这样：

```py
INSTALLED_APPS = ( 
'django.contrib.admin', 
'django.contrib.auth', 
'django.contrib.contenttypes', 
'django.contrib.sessions', 
'django.contrib.messages', 
'django.contrib.staticfiles', 
) 

```

要注册我们的`books`应用程序，请将`'books'`添加到`INSTALLED_APPS`中，以便设置最终看起来像这样（`'books'`指的是我们正在使用的`books`应用程序）：

```py
INSTALLED_APPS = ( 
'django.contrib.admin', 
'django.contrib.auth', 
'django.contrib.contenttypes', 
'django.contrib.sessions', 
'django.contrib.messages', 
'django.contrib.staticfiles', 
'books', 
) 

```

`INSTALLED_APPS`中的每个应用程序都由其完整的 Python 路径表示-即，由点分隔的导致应用程序包的路径。现在 Django 应用程序已在设置文件中激活，我们可以在数据库中创建数据库表。首先，让我们通过运行此命令来验证模型：

```py
python manage.py check

```

`check`命令运行 Django 系统检查框架-一组用于验证 Django 项目的静态检查。如果一切正常，您将看到消息`System check identified no issues (0 silenced)`。如果没有，请确保您正确输入了模型代码。错误输出应该为您提供有关代码错误的有用信息。每当您认为模型存在问题时，请运行`python manage.py check`。它往往会捕捉到所有常见的模型问题。

如果您的模型有效，请运行以下命令告诉 Django 您对模型进行了一些更改（在本例中，您创建了一个新模型）：

```py
python manage.py makemigrations books 

```

您应该看到类似以下内容的东西：

```py
Migrations for 'books': 
  0001_initial.py: 
   -Create model Author 
   -Create model Book 
   -Create model Publisher 
   -Add field publisher to book 

```

迁移是 Django 存储对模型的更改（因此是数据库模式）的方式-它们只是磁盘上的文件。在这种情况下，您将在`books`应用程序的`migrations`文件夹中找到名为`0001_initial.py`的文件。`migrate`命令将获取您的最新迁移文件并自动更新您的数据库模式，但首先让我们看看该迁移将运行的 SQL。`sqlmigrate`命令获取迁移名称并返回它们的 SQL：

```py
python manage.py sqlmigrate books 0001

```

你应该看到类似以下的内容（为了可读性重新格式化）：

```py
BEGIN; 

CREATE TABLE "books_author" ( 
    "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, 
    "first_name" varchar(30) NOT NULL, 
    "last_name" varchar(40) NOT NULL, 
    "email" varchar(254) NOT NULL 
); 
CREATE TABLE "books_book" ( 
    "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, 
    "title" varchar(100) NOT NULL, 
    "publication_date" date NOT NULL 
); 
CREATE TABLE "books_book_authors" ( 
    "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, 
    "book_id" integer NOT NULL REFERENCES "books_book" ("id"), 
    "author_id" integer NOT NULL REFERENCES "books_author" ("id"), 
    UNIQUE ("book_id", "author_id") 
); 
CREATE TABLE "books_publisher" ( 
    "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, 
    "name" varchar(30) NOT NULL, 
    "address" varchar(50) NOT NULL, 
    "city" varchar(60) NOT NULL, 
    "state_province" varchar(30) NOT NULL, 
    "country" varchar(50) NOT NULL, 
    "website" varchar(200) NOT NULL 
); 
CREATE TABLE "books_book__new" ( 
    "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, 
    "title" varchar(100) NOT NULL, 
    "publication_date" date NOT NULL, 
    "publisher_id" integer NOT NULL REFERENCES 
    "books_publisher" ("id") 
); 

INSERT INTO "books_book__new" ("id", "publisher_id", "title", 
"publication_date") SELECT "id", NULL, "title", "publication_date" FROM 
"books_book"; 

DROP TABLE "books_book"; 

ALTER TABLE "books_book__new" RENAME TO "books_book"; 

CREATE INDEX "books_book_2604cbea" ON "books_book" ("publisher_id"); 

COMMIT; 

```

请注意以下内容：

+   表名是通过组合应用程序的名称（`books`）和模型的小写名称（`publisher`，`book`和`author`）自动生成的。你可以覆盖这种行为，详细信息请参见附录 B，*数据库 API 参考*。

+   正如我们之前提到的，Django 会自动为每个表添加一个主键-`id`字段。你也可以覆盖这一点。按照惯例，Django 会将`"_id"`附加到外键字段名称。你可能已经猜到，你也可以覆盖这种行为。

+   外键关系通过`REFERENCES`语句明确表示。

这些`CREATE TABLE`语句是针对你正在使用的数据库定制的，因此数据库特定的字段类型，如`auto_increment`（MySQL），`serial`（PostgreSQL）或`integer primary key`（SQLite）都会自动处理。列名的引用也是一样的（例如，使用双引号或单引号）。这个示例输出是以 PostgreSQL 语法为例。

`sqlmigrate`命令实际上并不会创建表或者对数据库进行任何操作，它只是在屏幕上打印输出，这样你就可以看到如果要求 Django 执行的 SQL 是什么。如果你愿意，你可以将这些 SQL 复制粘贴到你的数据库客户端中，然而，Django 提供了一个更简单的方法将 SQL 提交到数据库：`migrate`命令：

```py
python manage.py migrate

```

运行该命令，你会看到类似以下的内容：

```py
Operations to perform:
 Apply all migrations: books
Running migrations:
 Rendering model states... DONE
 # ...
 Applying books.0001_initial... OK
 # ...

```

如果你想知道所有这些额外的内容是什么（在上面被注释掉的），第一次运行 migrate 时，Django 还会创建 Django 内置应用所需的所有系统表。迁移是 Django 传播你对模型所做更改（添加字段、删除模型等）到数据库模式的方式。它们被设计为大部分是自动的，但是也有一些注意事项。有关迁移的更多信息，请参见第二十一章，*高级数据库管理*。

# 基本数据访问

一旦你创建了一个模型，Django 会自动为这些模型提供一个高级别的 Python API。通过运行`python manage.py shell`并输入以下内容来尝试一下：

```py
>>> from books.models import Publisher 
>>> p1 = Publisher(name='Apress', address='2855 Telegraph Avenue', 
...     city='Berkeley', state_province='CA', country='U.S.A.', 
...     website='http://www.apress.com/') 
>>> p1.save() 
>>> p2 = Publisher(name="O'Reilly", address='10 Fawcett St.', 
...     city='Cambridge', state_province='MA', country='U.S.A.', 
...     website='http://www.oreilly.com/') 
>>> p2.save() 
>>> publisher_list = Publisher.objects.all() 
>>> publisher_list 
[<Publisher: Publisher object>, <Publisher: Publisher object>] 

```

这几行代码完成了很多事情。以下是重点：

+   首先，我们导入我们的`Publisher`模型类。这让我们可以与包含出版商的数据库表进行交互。

+   我们通过为每个字段实例化一个`Publisher`对象来创建一个`Publisher`对象-`name`，`address`等等。

+   要将对象保存到数据库中，请调用其`save()`方法。在幕后，Django 在这里执行了一个 SQL `INSERT`语句。

+   要从数据库中检索出出版商，使用属性`Publisher.objects`，你可以将其视为所有出版商的集合。使用语句`Publisher.objects.all()`获取数据库中所有`Publisher`对象的列表。在幕后，Django 在这里执行了一个 SQL `SELECT`语句。

有一件事值得一提，以防这个例子没有清楚地表明。当你使用 Django 模型 API 创建对象时，Django 不会将对象保存到数据库，直到你调用`save()`方法：

```py
p1 = Publisher(...) 
# At this point, p1 is not saved to the database yet! 
p1.save() 
# Now it is. 

```

如果你想要在一步中创建一个对象并将其保存到数据库中，可以使用`objects.create()`方法。这个例子等同于上面的例子：

```py
>>> p1 = Publisher.objects.create(name='Apress', 
...     address='2855 Telegraph Avenue', 
...     city='Berkeley', state_province='CA', country='U.S.A.', 
...     website='http://www.apress.com/') 
>>> p2 = Publisher.objects.create(name="O'Reilly", 
...     address='10 Fawcett St.', city='Cambridge', 
...     state_province='MA', country='U.S.A.', 
...     website='http://www.oreilly.com/') 
>>> publisher_list = Publisher.objects.all() 
>>> publisher_list 
[<Publisher: Publisher object>, <Publisher: Publisher object>] 

```

当然，你可以使用 Django 数据库 API 做很多事情，但首先，让我们解决一个小烦恼。

## 添加模型字符串表示

当我们打印出出版商列表时，我们得到的只是这种不太有用的显示，这使得很难区分`Publisher`对象：

```py
[<Publisher: Publisher object>, <Publisher: Publisher object>] 

```

我们可以通过在`Publisher`类中添加一个名为`__str__()`的方法来轻松解决这个问题。`__str__()`方法告诉 Python 如何显示对象的可读表示。通过为这三个模型添加`__str__()`方法，你可以看到它的作用。

```py
from django.db import models 

class Publisher(models.Model): 
    name = models.CharField(max_length=30) 
    address = models.CharField(max_length=50) 
    city = models.CharField(max_length=60) 
    state_province = models.CharField(max_length=30) 
    country = models.CharField(max_length=50) 
    website = models.URLField() 

 def __str__(self): 
 return self.name 

class Author(models.Model): 
    first_name = models.CharField(max_length=30) 
    last_name = models.CharField(max_length=40) 
    email = models.EmailField() 

 def __str__(self):
 return u'%s %s' % 
                                (self.first_name, self.last_name) 

class Book(models.Model): 
    title = models.CharField(max_length=100) 
    authors = models.ManyToManyField(Author) 
    publisher = models.ForeignKey(Publisher) 
    publication_date = models.DateField() 

 def __str__(self):
 return self.title

```

如您所见，`__str__()`方法可以根据需要执行任何操作，以返回对象的表示。在这里，`Publisher`和`Book`的`__str__()`方法分别返回对象的名称和标题，但`Author`的`__str__()`方法稍微复杂一些-它将`first_name`和`last_name`字段拼接在一起，用空格分隔。`__str__()`的唯一要求是返回一个字符串对象。如果`__str__()`没有返回一个字符串对象-如果它返回了一个整数-那么 Python 将引发一个类似于以下的`TypeError`消息：

```py
TypeError: __str__ returned non-string (type int). 

```

要使`__str__()`的更改生效，请退出 Python shell，然后使用`python manage.py shell`再次进入。 （这是使代码更改生效的最简单方法。）现在`Publisher`对象的列表更容易理解了：

```py
>>> from books.models import Publisher 
>>> publisher_list = Publisher.objects.all() 
>>> publisher_list 
[<Publisher: Apress>, <Publisher: O'Reilly>] 

```

确保您定义的任何模型都有一个`__str__()`方法-不仅是为了在使用交互式解释器时方便您自己，而且还因为 Django 在需要显示对象时使用`__str__()`的输出。最后，请注意，`__str__()`是向模型添加行为的一个很好的例子。Django 模型描述了对象的数据库表布局，还描述了对象知道如何执行的任何功能。`__str__()`就是这种功能的一个例子-模型知道如何显示自己。

## 插入和更新数据

您已经看到了这个操作：要向数据库插入一行数据，首先使用关键字参数创建模型的实例，如下所示：

```py
>>> p = Publisher(name='Apress', 
...         address='2855 Telegraph Ave.', 
...         city='Berkeley', 
...         state_province='CA', 
...         country='U.S.A.', 
...         website='http://www.apress.com/') 

```

正如我们上面所指出的，实例化模型类的行为并不会触及数据库。直到您调用`save()`，记录才会保存到数据库中，就像这样：

```py
>>> p.save() 

```

在 SQL 中，这大致可以翻译为以下内容：

```py
INSERT INTO books_publisher 
    (name, address, city, state_province, country, website) 
VALUES 
    ('Apress', '2855 Telegraph Ave.', 'Berkeley', 'CA', 
     'U.S.A.', 'http://www.apress.com/'); 

```

因为`Publisher`模型使用自增主键`id`，对`save()`的初始调用还做了一件事：它计算了记录的主键值，并将其设置为实例的`id`属性：

```py
>>> p.id 
52    # this will differ based on your own data 

```

对`save()`的后续调用将在原地保存记录，而不是创建新记录（即执行 SQL 的`UPDATE`语句而不是`INSERT`）：

```py
>>> p.name = 'Apress Publishing' 
>>> p.save() 

```

前面的`save()`语句将导致大致以下的 SQL：

```py
UPDATE books_publisher SET 
    name = 'Apress Publishing', 
    address = '2855 Telegraph Ave.', 
    city = 'Berkeley', 
    state_province = 'CA', 
    country = 'U.S.A.', 
    website = 'http://www.apress.com' 
WHERE id = 52; 

```

是的，请注意，所有字段都将被更新，而不仅仅是已更改的字段。根据您的应用程序，这可能会导致竞争条件。请参阅下面的*在一条语句中更新多个对象*，了解如何执行这个（略有不同）查询：

```py
UPDATE books_publisher SET 
    name = 'Apress Publishing' 
WHERE id=52; 

```

## 选择对象

了解如何创建和更新数据库记录是至关重要的，但很有可能您构建的 Web 应用程序将更多地查询现有对象，而不是创建新对象。我们已经看到了检索给定模型的每条记录的方法：

```py
>>> Publisher.objects.all() 
[<Publisher: Apress>, <Publisher: O'Reilly>] 

```

这大致对应于以下 SQL：

```py
SELECT id, name, address, city, state_province, country, website 
FROM books_publisher; 

```

### 注意

请注意，Django 在查找数据时不使用`SELECT *`，而是明确列出所有字段。这是有意设计的：在某些情况下，`SELECT *`可能会更慢，并且（更重要的是）列出字段更贴近 Python 之禅的一个原则：*明确胜于隐晦*。有关 Python 之禅的更多信息，请尝试在 Python 提示符下输入`import this`。

让我们仔细看看`Publisher.objects.all()`这行的每个部分：

+   首先，我们有我们定义的模型`Publisher`。这里没有什么意外：当您想要查找数据时，您使用该数据的模型。

+   接下来，我们有`objects`属性。这被称为**管理器**。管理器在第九章*高级模型*中有详细讨论。现在，您需要知道的是，管理器负责处理数据的所有*表级*操作，包括最重要的数据查找。所有模型都会自动获得一个`objects`管理器；每当您想要查找模型实例时，都会使用它。

+   最后，我们有`all()`。这是`objects`管理器上的一个方法，它返回数据库中的所有行。虽然这个对象看起来像一个列表，但它实际上是一个**QuerySet**-一个表示数据库中特定一组行的对象。附录 C，*通用视图参考*，详细介绍了 QuerySets。在本章的其余部分，我们将把它们当作它们模拟的列表来处理。

任何数据库查找都会遵循这个一般模式-我们将在我们想要查询的模型上调用附加的管理器的方法。

## 过滤数据

自然地，很少有人希望一次从数据库中选择所有内容；在大多数情况下，您将希望处理您数据的一个子集。在 Django API 中，您可以使用`filter()`方法过滤您的数据：

```py
>>> Publisher.objects.filter(name='Apress') 
[<Publisher: Apress>] 

```

`filter()`接受关键字参数，这些参数被转换为适当的 SQL `WHERE`子句。前面的例子将被转换为类似于这样的东西：

```py
SELECT id, name, address, city, state_province, country, website 
FROM books_publisher 
WHERE name = 'Apress'; 

```

您可以将多个参数传递给`filter()`以进一步缩小范围：

```py
>>> Publisher.objects.filter(country="U.S.A.", state_province="CA") 
[<Publisher: Apress>] 

```

这些多个参数被转换为 SQL `AND`子句。因此，代码片段中的示例被转换为以下内容：

```py
SELECT id, name, address, city, state_province, country, website 
FROM books_publisher 
WHERE country = 'U.S.A.' 
AND state_province = 'CA'; 

```

请注意，默认情况下，查找使用 SQL `=`运算符进行精确匹配查找。其他查找类型也是可用的：

```py
>>> Publisher.objects.filter(name__contains="press") 
[<Publisher: Apress>] 

```

在`name`和`contains`之间有一个双下划线。像 Python 本身一样，Django 使用双下划线来表示发生了一些魔术-这里，`__contains`部分被 Django 转换为 SQL `LIKE`语句：

```py
SELECT id, name, address, city, state_province, country, website 
FROM books_publisher 
WHERE name LIKE '%press%'; 

```

还有许多其他类型的查找可用，包括`icontains`（不区分大小写的`LIKE`）、`startswith`和`endswith`，以及`range`（SQL `BETWEEN`查询）。附录 C，*通用视图参考*，详细描述了所有这些查找类型。

## 检索单个对象

上面的所有`filter()`示例都返回了一个`QuerySet`，您可以像对待列表一样对待它。有时，只获取单个对象比获取列表更方便。这就是`get()`方法的用途：

```py
>>> Publisher.objects.get(name="Apress") 
<Publisher: Apress> 

```

而不是返回一个列表（`QuerySet`），只返回一个单一对象。因此，导致多个对象的查询将引发异常：

```py
>>> Publisher.objects.get(country="U.S.A.") 
Traceback (most recent call last): 
    ... 
MultipleObjectsReturned: get() returned more than one Publisher -- it returned 2! Lookup parameters were {'country': 'U.S.A.'} 

```

返回没有对象的查询也会引发异常：

```py
>>> Publisher.objects.get(name="Penguin") 
Traceback (most recent call last): 
    ... 
DoesNotExist: Publisher matching query does not exist. 

```

`DoesNotExist`异常是模型类`Publisher.DoesNotExist`的属性。在您的应用程序中，您将希望捕获这些异常，就像这样：

```py
try: 
    p = Publisher.objects.get(name='Apress') 
except Publisher.DoesNotExist: 
    print ("Apress isn't in the database yet.") 
else: 
    print ("Apress is in the database.") 

```

## 排序数据

当您尝试之前的示例时，您可能会发现对象以看似随机的顺序返回。您没有想象的事情；到目前为止，我们还没有告诉数据库如何对其结果进行排序，因此我们只是以数据库选择的某种任意顺序返回数据。在您的 Django 应用程序中，您可能希望根据某个值-比如按字母顺序-对结果进行排序。要做到这一点，请使用`order_by()`方法：

```py
>>> Publisher.objects.order_by("name") 
[<Publisher: Apress>, <Publisher: O'Reilly>] 

```

这看起来与之前的`all()`示例没有太大不同，但是现在的 SQL 包括了特定的排序：

```py
SELECT id, name, address, city, state_province, country, website 
FROM books_publisher 
ORDER BY name; 

```

您可以按任何您喜欢的字段排序：

```py
>>> Publisher.objects.order_by("address") 
 [<Publisher: O'Reilly>, <Publisher: Apress>] 

>>> Publisher.objects.order_by("state_province") 
 [<Publisher: Apress>, <Publisher: O'Reilly>] 

```

要按多个字段排序（其中第二个字段用于消除第一个字段相同时的排序），请使用多个参数：

```py
>>> Publisher.objects.order_by("state_province", "address") 
 [<Publisher: Apress>, <Publisher: O'Reilly>] 

```

您还可以通过在字段名前加上“-”（减号）来指定反向排序：

```py
>>> Publisher.objects.order_by("-name") 
[<Publisher: O'Reilly>, <Publisher: Apress>] 

```

虽然这种灵活性很有用，但是一直使用`order_by()`可能会相当重复。大多数情况下，您通常会有一个特定的字段，您希望按照它进行排序。在这些情况下，Django 允许您在模型中指定默认排序：

```py
class Publisher(models.Model): 
    name = models.CharField(max_length=30) 
    address = models.CharField(max_length=50) 
    city = models.CharField(max_length=60) 
    state_province = models.CharField(max_length=30) 
    country = models.CharField(max_length=50) 
    website = models.URLField() 

    def __str__(self): 
        return self.name 

    class Meta:
 ordering = ['name']

```

在这里，我们介绍了一个新概念：`class Meta`，它是嵌入在`Publisher`类定义中的类（也就是说，它是缩进在`class Publisher`内部的）。您可以在任何模型上使用这个`Meta`类来指定各种特定于模型的选项。`Meta`选项的完整参考可在附录 B 中找到，但现在我们关注的是排序选项。如果您指定了这个选项，它告诉 Django，除非使用`order_by()`明确给出排序，否则所有`Publisher`对象在使用 Django 数据库 API 检索时都应该按`name`字段排序。

## 链接查找

您已经看到了如何过滤数据，也看到了如何对其进行排序。当然，通常情况下，您需要同时做这两件事。在这些情况下，您只需将查找链接在一起：

```py
>>> Publisher.objects.filter(country="U.S.A.").order_by("-name") 
[<Publisher: O'Reilly>, <Publisher: Apress>] 

```

正如您所期望的，这会转换为一个同时具有`WHERE`和`ORDER BY`的 SQL 查询：

```py
SELECT id, name, address, city, state_province, country, website 
FROM books_publisher 
WHERE country = 'U.S.A' 
ORDER BY name DESC; 

```

## 切片数据

另一个常见的需求是仅查找固定数量的行。想象一下，您的数据库中有成千上万的出版商，但您只想显示第一个。您可以使用 Python 的标准列表切片语法来实现：

```py
>>> Publisher.objects.order_by('name')[0] 
<Publisher: Apress> 

```

这大致对应于：

```py
SELECT id, name, address, city, state_province, country, website 
FROM books_publisher 
ORDER BY name 
LIMIT 1; 

```

类似地，您可以使用 Python 的范围切片语法检索特定的数据子集：

```py
>>> Publisher.objects.order_by('name')[0:2] 

```

这返回两个对象，大致翻译为：

```py
SELECT id, name, address, city, state_province, country, website 
FROM books_publisher 
ORDER BY name 
OFFSET 0 LIMIT 2; 

```

请注意，不支持负切片：

```py
>>> Publisher.objects.order_by('name')[-1] 
Traceback (most recent call last): 
  ... 
AssertionError: Negative indexing is not supported. 

```

不过，这很容易解决。只需更改`order_by()`语句，就像这样：

```py
>>> Publisher.objects.order_by('-name')[0] 

```

## 在一个语句中更新多个对象

我们在*插入和更新数据*部分指出，模型`save()`方法会更新行中的所有列。根据您的应用程序，您可能只想更新部分列。例如，假设我们要更新 Apress `Publisher`将名称从`'Apress'`更改为`'Apress Publishing'`。使用`save()`，它看起来会像这样：

```py
>>> p = Publisher.objects.get(name='Apress') 
>>> p.name = 'Apress Publishing' 
>>> p.save() 

```

这大致对应以下 SQL：

```py
SELECT id, name, address, city, state_province, country, website 
FROM books_publisher 
WHERE name = 'Apress'; 

UPDATE books_publisher SET 
    name = 'Apress Publishing', 
    address = '2855 Telegraph Ave.', 
    city = 'Berkeley', 
    state_province = 'CA', 
    country = 'U.S.A.', 
    website = 'http://www.apress.com' 
WHERE id = 52; 

```

（请注意，此示例假定 Apress 的出版商 ID 为`52`。）您可以在此示例中看到，Django 的`save()`方法设置了所有列的值，而不仅仅是`name`列。如果您处于其他列可能由于其他进程而发生变化的环境中，最好只更改您需要更改的列。要做到这一点，请在`QuerySet`对象上使用`update()`方法。以下是一个例子：

```py
>>> Publisher.objects.filter(id=52).update(name='Apress Publishing') 

```

这里的 SQL 转换效率更高，没有竞争条件的机会：

```py
UPDATE books_publisher 
SET name = 'Apress Publishing' 
WHERE id = 52; 

```

`update()`方法适用于任何`QuerySet`，这意味着您可以批量编辑多条记录。以下是您可能如何更改每个`Publisher`记录中的`country`从`'U.S.A.'`更改为`USA`：

```py
>>> Publisher.objects.all().update(country='USA') 
2 

```

`update()`方法有一个返回值-表示更改了多少条记录的整数。在上面的例子中，我们得到了`2`。

## 删除对象

要从数据库中删除对象，只需调用对象的`delete()`方法：

```py
>>> p = Publisher.objects.get(name="O'Reilly") 
>>> p.delete() 
>>> Publisher.objects.all() 
[<Publisher: Apress Publishing>] 

```

您还可以通过在任何`QuerySet`的结果上调用`delete()`来批量删除对象。这类似于我们在上一节中展示的`update()`方法：

```py
>>> Publisher.objects.filter(country='USA').delete() 
>>> Publisher.objects.all().delete() 
>>> Publisher.objects.all() 
[] 

```

小心删除您的数据！为了防止删除特定表中的所有数据，Django 要求您明确使用`all()`，如果要删除表中的所有内容。例如，这样是行不通的：

```py
>>> Publisher.objects.delete() 
Traceback (most recent call last): 
  File "", line 1, in  
AttributeError: 'Manager' object has no attribute 'delete' 

```

但如果添加`all()`方法，它将起作用：

```py
>>> Publisher.objects.all().delete() 

```

如果您只是删除数据的一个子集，您不需要包括`all()`。重复之前的例子：

```py
>>> Publisher.objects.filter(country='USA').delete() 

```

# 接下来是什么？

阅读完本章后，您已经掌握了足够的 Django 模型知识，可以编写基本的数据库应用程序。第九章，“高级模型”，将提供有关 Django 数据库层更高级用法的一些信息。一旦您定义了模型，下一步就是向数据库填充数据。您可能有遗留数据，这种情况下第二十一章，“高级数据库管理”，将为您提供有关与遗留数据库集成的建议。您可能依赖站点用户提供数据，这种情况下第六章，“表单”，将教您如何处理用户提交的表单数据。但在某些情况下，您或您的团队可能需要手动输入数据，这种情况下拥有一个基于 Web 的界面来输入和管理数据将非常有帮助。下一章将介绍 Django 的管理界面，它正是为了这个目的而存在的。
